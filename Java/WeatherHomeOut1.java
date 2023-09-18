package hello;
import java.util.Scanner;
public class WeatherHomeOut1 {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.println("今天是晴天還是陰天");
		String wheather = input.next();
		if(wheather=="晴天") {
			System.out.println("要散步還是逛街");
			String walkBuy = input.next();
			if(walkBuy=="散步") {
				System.out.println("今天是晴天去散步");
			}else {
				System.out.println("今天是晴天去逛街");
			}
		}else {
			System.out.println("要睡覺還是看影片");
			String sleepVideo = input.next();
			if(sleepVideo == "睡覺") {
				System.out.println("今天陰天在家睡覺");
			}else {
				System.out.println("今天陰天在家看影片");
			}
		}
	}
}
