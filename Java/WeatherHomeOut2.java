package hello;
import java.util.Scanner;
public class WeatherHomeOut2 {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.println("今天是晴天還是陰天");
		String wheather = input.next();
		sunWindy(wheather);
	}
public static void sunWindy(String wheather) {
	if(wheather=="晴天") {
		System.out.println("要去散步還是逛街");
	}else{
		System.out.println("要睡覺還是看電影");
	}
}
}
